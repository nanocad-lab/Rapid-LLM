# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Tuple

import numpy as np


class Base:
    def __init__(self, exp_config):
        self.exp_config = exp_config
        self._precision_cfg = exp_config.sw_config.precision
        self.precision = self._precision_cfg.activations
        self.proc_chip_area_budget = exp_config.area_breakdown.proc_chip_area_budget
        self.TDP = exp_config.power_breakdown.TDP
        self.throughput = -1
        node_width = math.sqrt(self.proc_chip_area_budget)
        self.core_perimeter = node_width * 4

    def get_throughput(self):
        assert self.throughput != -1
        return self.throughput

    def solve_poly(self, p0, p1, p2, p3):
        # solve p0.x^3 + p1.x^2 + p2.x + p3 = 0
        roots = np.roots([p0, p1, p2, p3])
        real_roots = roots.real[
            abs(roots.imag) < 1e-10
        ]  # where I chose 1-e10 as a threshold
        return real_roots[0]


class Memory(Base):
    def __init__(self, exp_config, level, core):
        super().__init__(exp_config)
        self.size = -1
        self.tile_dim = -1
        self.latency = -1
        if core is None:
            raise ValueError("Memory requires a Core instance")
        self.core = core
        self.level = level

    def get_tile_dim(self):
        assert self.tile_dim != -1
        return self.tile_dim

    def get_latency(self):
        assert self.latency != -1
        return self.latency

    def get_tile_dims(self):
        return self.get_power2_tile_dims()

    def calc_waves_per_sm(self, M, K, N, m, k, n):
        bytes_required = self.precision * (m * k + k * n)
        grid_size = math.ceil(M / m) * math.ceil(N / n)

        compute_time = 2 * (m * n * k) / 312e12
        load_time = (self.precision * (m * k + k * n)) / (7050 * 1024 ** 3)

        stages = 3 if load_time < compute_time else 5
        smem_required = bytes_required * stages
        if smem_required > self.size_per_bundle:
            return -1
        
        reg_accum = m * n
        reg_input = 16 * (m + n)
        if reg_accum + reg_input > 65536:
            return -1

        smem_blocks = self.size_per_bundle // smem_required
        reg_blocks = 65536 // (reg_accum + reg_input)

        return grid_size / min(smem_blocks, reg_blocks) / 108


    def get_gemm_based_tile_dims(self, M, K, N):
        m_dims = [ M >> i for i in range(M.bit_length()) if (M >> i) >= 1 ]
        k_dims = [ K >> i for i in range(K.bit_length()) if (K >> i) >= 1 ]
        n_dims = [ N >> i for i in range(N.bit_length()) if (N >> i) >= 1 ]

        valid_tiles = []
        for m in m_dims[::-1]:
            for k in k_dims[::-1]:
                for n in n_dims[::-1]:
                    bytes_required = self.precision * (m * k + k * n + m * n)
                    if bytes_required <= self.size_per_bundle:
                        valid_tiles.append((m, k, n))
        final_tiles = set()
        for candidate in valid_tiles:
            is_dominated = False
            for tile in final_tiles:
                m, k, n = candidate
                if ((m, k) == tile[:2] and n < tile[2]
                    or (k, n) == tile[1:] and m < tile[0]
                    or (m, n) == (tile[0], tile[2]) and k < tile[1]):
                    is_dominated = True
                    break

            if not is_dominated:
                to_remove = {
                    tile for tile in final_tiles
                    if ((m, k) == tile[:2] and n > tile[2]
                        or (k, n) == tile[1:] and m > tile[0]
                        or (m, n) == (tile[0], tile[2]) and k > tile[1])
                }
                final_tiles.difference_update(to_remove)
                final_tiles.add(candidate)
        return final_tiles

    def get_power2_tile_dims(self):
        np.random.seed(1)
        tile_dim_candidates = set()
        num_candidates = 20
        M = self.size_per_bundle / self.precision
        max_power = int(math.floor(math.log2(M)))

        self.calc_tile_dim()
        square_tile = self.get_tile_dim()
        tile_dim_candidates.add((square_tile, square_tile, square_tile))
        tile_dim_candidates.add((square_tile // 2, square_tile, square_tile * 2))
        while len(tile_dim_candidates) < num_candidates:
            z = -1
            while z < 0:
                s = [pow(2, i) for i in np.random.randint(0, max_power, 2)]
                # store goes through cache at level 0 and 1 (register and shared memory)
                assert self.level >= 0 and self.level <= 3
                if self.level <= 1:
                    z = math.floor((M - s[0] * s[1]) / (s[0] + s[1]))
                else:
                    # store bypasses cache, directly goes to memory
                    z = math.floor((M - s[0] * s[1]) / s[1])

                if z <= 0:
                    continue

                z = int(math.pow(2, math.floor(math.log2(z))))
                tile_dim = (s[0], s[1], z)
                tile_dim_candidates.add(tile_dim)

        return list(tile_dim_candidates)

    def calc_tile_dim(self):
        self.tile_dim = 0

        if self.scope == "global":
            divisor = 1
        elif self.scope == "mcu-bundle":
            divisor = self.core.num_bundle
        elif self.scope == "mcu":
            divisor = self.core.num_mcu
        else:
            raise NotImplementedError()

        self.size_per_bundle = 0 if (divisor == 0) else self.size / divisor

        if self.size > 0:
            self.tile_dim = math.ceil(
                math.pow(
                    2,
                    math.floor(
                        math.log(
                            math.sqrt((self.size_per_bundle / self.precision) / 2), 2
                        )
                    ),
                )
            )
            # self.tile_dim = math.floor(math.sqrt((self.size_per_bundle / self.precision) / 3))


class Core(Base):
    def __init__(self, exp_config):
        super().__init__(exp_config)
        self.tot_power = exp_config.power_breakdown.core  # * self.TDP
        self.tot_area = exp_config.area_breakdown.core  # * self.proc_chip_area_budget

        self.FMA_dims = exp_config.tech_config.core.FMA_dims
        self.dataflow = exp_config.tech_config.core.dataflow

        self.nominal_flop_rate_per_mcu = exp_config.tech_config.core.nominal_flop_rate_per_mcu # TODO: Define it as a function of precision
        self.nominal_energy_per_flop = exp_config.tech_config.core.nominal_energy_per_flop
        self.nominal_power_per_mcu = exp_config.tech_config.core.nominal_power_per_mcu
        self.util = exp_config.tech_config.core.util

        # self.operating_voltage            = exp_config.tech_config.core.operating_voltage
        self.nominal_voltage = exp_config.tech_config.core.nominal_voltage
        self.threshold_voltage = exp_config.tech_config.core.threshold_voltage
        self.margin_voltage = exp_config.tech_config.core.margin_voltage

        # Assumption: performance scales linearly with area
        self.operating_area_per_mcu = exp_config.tech_config.core.operating_area_per_mcu
        if exp_config.tech_config.core.nominal_area_per_mcu:
            self.nominal_area_per_mcu = exp_config.tech_config.core.nominal_area_per_mcu
            self.area_scaling = self.operating_area_per_mcu / self.nominal_area_per_mcu
        else:
            self.area_scaling = 1
            self.nominal_area_per_mcu = self.operating_area_per_mcu

        self.num_mcu_per_bundle = exp_config.tech_config.core.num_mcu_per_bundle
        if exp_config.tech_config.core.num_bundles:
            self.num_bundle = exp_config.tech_config.core.num_bundles
            self.num_mcu = self.num_bundle * self.num_mcu_per_bundle
        else:
            self.num_mcu = int(self.tot_area // self.operating_area_per_mcu)
            self.num_bundle = int(self.num_mcu // self.num_mcu_per_bundle)

        self.operating_flop_rate_per_mcu = self.nominal_flop_rate_per_mcu * self.area_scaling
        self.nominal_power = self.nominal_power_per_mcu * self.num_mcu * self.area_scaling
        
        if self.tot_power > 0 and self.nominal_power > 0:
            self.calc_operating_voltage()
            if exp_config.tech_config.core.operating_frequency:
                self.operating_freq = exp_config.tech_config.core.operating_frequency
                self.operating_power_per_mcu = self.tot_power / self.num_mcu
            elif exp_config.tech_config.core.nominal_frequency:
                self.nominal_freq = exp_config.tech_config.core.nominal_frequency
                self.calc_operating_frequency()
        else:
            self.operating_freq = 0

        self.calc_throughput()

    def calc_operating_voltage(self):
        # minimum voltage that meets power constraints
        self.operating_voltage = self.solve_poly(
            p0=1,
            p1=-2 * self.threshold_voltage,
            p2=self.threshold_voltage**2,
            p3=-1
            * self.tot_power
            / self.nominal_power
            * self.nominal_voltage
            * (self.nominal_voltage - self.threshold_voltage) ** 2,
        )

        self.frequency_scaling_factor = 1
        if self.operating_voltage < (self.threshold_voltage + self.margin_voltage):
            self.scaled_voltage = self.threshold_voltage + self.margin_voltage
            self.frequency_scaling_factor = (
                self.operating_voltage / self.scaled_voltage
            ) ** 2
            self.operating_voltage = self.scaled_voltage

    def calc_operating_frequency(self):
        # Calculate operating frequency at minimum voltage
        self.operating_freq = self.nominal_freq * (
            (
                (self.operating_voltage - self.threshold_voltage) ** 2
                / (self.operating_voltage)
            )
            / (
                (self.nominal_voltage - self.threshold_voltage) ** 2
                / self.nominal_voltage
            )
        )

        self.operating_freq = self.frequency_scaling_factor * self.operating_freq

        self.operating_power_per_mcu = (
            self.nominal_power_per_mcu
            * (self.operating_freq / self.nominal_freq)
            * (self.operating_voltage / self.nominal_voltage) ** 2
        )

    def calc_throughput(self):
        self.operating_throughput = self.operating_flop_rate_per_mcu * self.operating_freq * self.num_mcu
        self.throughput = self.operating_throughput * self.util


class MemoryHierarchy(Base):
    def __init__(self, exp_config, *, core=None):
        super().__init__(exp_config)
        self.core = core or Core(exp_config)
        self.num_levels = exp_config.memory_hierarchy.num_levels
        self.mem_layer = [None] * self.num_levels

        for level in range(0, self.num_levels):
            mem_config = exp_config.memory_hierarchy.mem_hr[level]

            if mem_config.type == "DRAM":
                self.mem_layer[level] = DRAM(exp_config, mem_config, level, self.core)
            elif mem_config.type == "SRAM-R":
                self.mem_layer[level] = SRAM(
                    exp_config,
                    exp_config.power_breakdown.reg_mem / exp_config.power_breakdown.TDP,
                    exp_config.area_breakdown.reg_mem
                    / exp_config.area_breakdown.proc_chip_area_budget,
                    exp_config.tech_config.SRAMR,
                    mem_config,
                    level,
                    self.core,
                )
            elif mem_config.type == "SRAM-L1":
                self.mem_layer[level] = SRAM(
                    exp_config,
                    exp_config.power_breakdown.L1 / exp_config.power_breakdown.TDP,
                    exp_config.area_breakdown.L1
                    / exp_config.area_breakdown.proc_chip_area_budget,
                    exp_config.tech_config.SRAML1,
                    mem_config,
                    level,
                    self.core,
                )
            elif mem_config.type == "SRAM-L2":
                self.mem_layer[level] = SRAM(
                    exp_config,
                    exp_config.power_breakdown.L2 / exp_config.power_breakdown.TDP,
                    exp_config.area_breakdown.L2
                    / exp_config.area_breakdown.proc_chip_area_budget,
                    exp_config.tech_config.SRAML2,
                    mem_config,
                    level,
                    self.core,
                )
            else:
                NotImplemented()


class DRAM(Memory):
    def __init__(self, exp_config, mem_config, level, core):
        super().__init__(exp_config, level, core)
        self.tot_power = exp_config.power_breakdown.DRAM  # * self.TDP
        self.tot_area = exp_config.area_breakdown.node_area_budget - self.proc_chip_area_budget
        self.tot_mem_ctrl_area = exp_config.area_breakdown.DRAM  # * self.proc_chip_area_budget
        self.mem_ctrl_area = exp_config.tech_config.DRAM.mem_ctrl_area
        self.dynamic_energy_per_bit = exp_config.tech_config.DRAM.dynamic_energy_per_bit
        self.static_power_per_byte = exp_config.tech_config.DRAM.static_power_per_bit * 8
        self.area_per_byte = exp_config.tech_config.DRAM.area_per_bit * 8
        self.stack_capacity = exp_config.tech_config.DRAM.stack_capacity
        self.area_per_stack = exp_config.tech_config.DRAM.area_per_stack
        self.latency = exp_config.tech_config.DRAM.latency
        self.scope = mem_config.scope
        self.util = exp_config.tech_config.DRAM.util
        
        if exp_config.tech_config.DRAM.nominal_frequency:
            self.nominal_freq = exp_config.tech_config.DRAM.nominal_frequency
        elif exp_config.tech_config.DRAM.operating_frequency:
            self.nominal_freq = exp_config.tech_config.DRAM.operating_frequency
        else:
            self.nominal_freq = 0.1

        self.nominal_voltage = exp_config.tech_config.DRAM.nominal_voltage
        self.threshold_voltage = exp_config.tech_config.DRAM.threshold_voltage
        self.margin_voltage = exp_config.tech_config.DRAM.margin_voltage
        self.max_voltage = exp_config.tech_config.DRAM.max_voltage

        if exp_config.tech_config.DRAM.num_stacks:
            self.num_stacks = exp_config.tech_config.DRAM.num_stacks
        else:
            self.num_stacks = int(
                min(
                    self.tot_area // self.area_per_stack,
                    self.tot_mem_ctrl_area // self.mem_ctrl_area,
                )
            )

        self.num_links_per_mm = exp_config.tech_config.DRAM.num_links_per_mm
        self.num_links_per_stack = exp_config.tech_config.DRAM.num_links_per_stack

        self.perimeter_bound = int(
            self.core_perimeter
            * exp_config.perimeter_breakdown.DRAM
            * self.num_links_per_mm
        )
        self.num_links = min(
            self.perimeter_bound, self.num_links_per_stack * self.num_stacks
        )

        self.size = exp_config.tech_config.DRAM.size
        self.dynamic_throughput = exp_config.tech_config.DRAM.bandwidth

        self.calc_size()
        self.calc_active_energy()

        self.nominal_power = self.dynamic_energy_per_bit * self.num_links * self.nominal_freq

        if self.dynamic_power > 0 and self.nominal_power > 0:
            self.calc_operating_voltage()
            if exp_config.tech_config.DRAM.operating_frequency:
                self.operating_freq = exp_config.tech_config.DRAM.operating_frequency
            else:
                self.calc_operating_frequency()
        else:
            self.operating_freq = 0

        # if self.dynamic_power > 0 and self.nominal_power > 0:
        #     self.calcOperatingVoltageFrequency()
        # else:
        #     self.operating_freq = 0

        self.calc_throughput()

        if self.dynamic_throughput <= 0:
            assert self.dynamic_throughput == 0
            self.num_stacks = 0

        self.calc_size()
        self.calc_tile_dim()

    def calc_operating_voltage(self):
        self.operating_voltage = self.solve_poly(
            p0=1,
            p1=-2 * self.threshold_voltage,
            p2=self.threshold_voltage**2,
            p3=-1
            * self.dynamic_power
            / self.nominal_power
            * self.nominal_voltage
            * (self.nominal_voltage - self.threshold_voltage) ** 2,
        )

        self.frequency_scaling_factor = 1
        self.max_freq = None
        if self.operating_voltage < (self.threshold_voltage + self.margin_voltage):
            self.scaled_voltage = self.threshold_voltage + self.margin_voltage
            self.frequency_scaling_factor = (self.operating_voltage / self.scaled_voltage) ** 2
            self.operating_voltage = self.scaled_voltage
        elif self.operating_voltage > self.max_voltage:
            self.max_freq = self.operating_freq * (
                ((self.max_voltage - self.threshold_voltage) ** 2 / (self.max_voltage))
                / (
                    (self.operating_voltage - self.threshold_voltage) ** 2
                    / self.operating_voltage
                )
            )
            self.operating_voltage = self.max_voltage

    def calc_operating_frequency(self):
        # operating frequency at minimum voltage
        self.operating_freq = self.nominal_freq * (
            (
                (self.operating_voltage - self.threshold_voltage) ** 2
                / (self.operating_voltage)
            )
            / (
                (self.nominal_voltage - self.threshold_voltage) ** 2
                / self.nominal_voltage
            )
        )
        self.operating_freq = self.frequency_scaling_factor * self.operating_freq
        if self.max_freq:
            self.operating_freq = self.max_freq

    def calc_active_energy(self):
        self.dynamic_power = (
            0
            if (self.tot_power < self.static_power_per_byte * self.size)
            else (self.tot_power - self.static_power_per_byte * self.size)
        )

    def calc_throughput(self):
        if not self.dynamic_throughput:
            self.dynamic_throughput = (
                0 if (self.size == 0) else self.num_links * self.operating_freq / 8
            )
        self.stack_bw = (
            0 if self.num_stacks == 0 else self.dynamic_throughput / self.num_stacks
        )
        self.throughput = self.dynamic_throughput * self.util

    def calc_size(self):
        # self.nominal_throughput         = self.tot_power / self.dynamic_energy_per_byte
        # self.size                       = min((self.nominal_throughput / self.stack_bw) * self.stack_capacity,
        #                                         self.cell_area / self.area_per_byte)
        if self.size:
            self.num_stacks = self.size // self.stack_capacity
            self.num_links = self.num_links_per_stack * self.num_stacks
        else:
            self.size = self.num_stacks * self.stack_capacity



class SRAM(Memory):
    def __init__(
        self,
        exp_config,
        power_config,
        area_config,
        tech_config,
        mem_hierarchy_config,
        level,
        core,
    ):
        super().__init__(exp_config, level, core)
        self.tot_power = power_config * self.TDP
        self.tot_area = area_config * self.proc_chip_area_budget
        self.dynamic_energy_per_bit = tech_config.dynamic_energy_per_bit
        self.dynamic_energy_per_byte = self.dynamic_energy_per_bit * 8
        self.static_power_per_byte = tech_config.static_power_per_bit * 8
        self.area_per_byte = tech_config.area_per_bit * 8
        self.bank_capacity = tech_config.bank_capacity
        self.controller_area_per_link = tech_config.controller_area_per_link
        self.latency = tech_config.latency
        self.overhead = (
            tech_config.overhead
        )  # percetage of cells dedicated to cicuitry overhead for SRAM cells
        self.cell_percentage = 1 - self.overhead
        self.util = tech_config.util
        self.scope = mem_hierarchy_config.scope
        self.type = mem_hierarchy_config.type
        self.bank_area = self.bank_capacity * self.area_per_byte
        self.num_banks = int(
            math.floor(
                (self.cell_percentage * self.tot_area)
                // (
                    self.bank_area
                    + self.cell_percentage
                    * self.core.num_bundle
                    * self.controller_area_per_link
                )
            )
        )

        self.size = tech_config.size
        self.dynamic_throughput = tech_config.bandwidth

        self.calc_size()

        self.calc_area()
        self.calc_active_energy()
        self.calc_throughput()

        if self.dynamic_throughput <= 0:
            self.num_banks = 0

        self.calc_size()
        self.calc_tile_dim()

    def calc_area(self):
        self.overhead_area = (
            self.num_banks * self.core.num_bundle * self.controller_area_per_link
        )
        self.cell_area = (self.tot_area - self.overhead_area) * self.cell_percentage
        if self.overhead_area > self.tot_area:
            self.cell_area = 0

    def calc_size(self):
        if self.size:
            self.num_banks = self.size // self.bank_capacity
        else:
            self.size = self.num_banks * self.bank_capacity

    def calc_active_energy(self):
        self.static_power = self.static_power_per_byte * self.size
        self.dynamic_power = (
            0
            if (self.tot_power < self.static_power)
            else (self.tot_power - self.static_power)
        )

    def calc_throughput(self):
        if not self.dynamic_throughput:
            self.dynamic_throughput = (
                0
                if (self.num_banks == 0)
                else self.dynamic_power / self.dynamic_energy_per_byte
            )
        self.throughput = self.dynamic_throughput * self.util

        self.bank_bw = (
            0 if (self.num_banks == 0) else self.dynamic_throughput / self.num_banks
        )


class Network(Base):
    def __init__(self, exp_config):
        super().__init__(exp_config)
        layout = getattr(exp_config, "network_layout", None)
        if layout is None:
            raise ValueError("hardware config is missing network layout information")
        self.layout = layout
        self.energies_per_bit = [
            float(getattr(dim, "energy_per_bit", 0.0))
            for dim in layout.dimensions
        ]

    def calc_throughput(self):
        primary = self.layout.primary_dimension()
        if primary is None:
            return 0.0, 0.0
        bw = float(primary.effective_bandwidth)
        return bw, bw

    def calc_latency(self):
        primary = self.layout.primary_dimension()
        if primary is None:
            return 0.0, 0.0
        latency = float(primary.latency)
        return latency, latency

    def get_link(self, parallelism: str) -> Tuple[float, float]:
        bandwidth, latency = self.layout.link_for_parallelism(parallelism)
        return float(bandwidth), float(latency)
